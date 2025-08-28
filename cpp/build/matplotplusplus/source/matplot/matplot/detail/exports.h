
#ifndef MATPLOT_EXPORTS_H
#define MATPLOT_EXPORTS_H

#ifdef SHARED_EXPORTS_BUILT_AS_STATIC
#  define MATPLOT_EXPORTS
#  define MATPLOT_NO_EXPORT
#else
#  ifndef MATPLOT_EXPORTS
#    ifdef matplot_EXPORTS
        /* We are building this library */
#      define MATPLOT_EXPORTS 
#    else
        /* We are using this library */
#      define MATPLOT_EXPORTS 
#    endif
#  endif

#  ifndef MATPLOT_NO_EXPORT
#    define MATPLOT_NO_EXPORT 
#  endif
#endif

#ifndef MATPLOT_DEPRECATED
#  define MATPLOT_DEPRECATED __declspec(deprecated)
#endif

#ifndef MATPLOT_DEPRECATED_EXPORT
#  define MATPLOT_DEPRECATED_EXPORT MATPLOT_EXPORTS MATPLOT_DEPRECATED
#endif

#ifndef MATPLOT_DEPRECATED_NO_EXPORT
#  define MATPLOT_DEPRECATED_NO_EXPORT MATPLOT_NO_EXPORT MATPLOT_DEPRECATED
#endif

/* NOLINTNEXTLINE(readability-avoid-unconditional-preprocessor-if) */
#if 0 /* DEFINE_NO_DEPRECATED */
#  ifndef MATPLOT_NO_DEPRECATED
#    define MATPLOT_NO_DEPRECATED
#  endif
#endif

#endif /* MATPLOT_EXPORTS_H */
